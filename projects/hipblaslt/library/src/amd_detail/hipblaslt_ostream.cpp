/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// Predeclare hipblaslt_abort_once() for friend declaration in hipblaslt_ostream.hpp
static void hipblaslt_abort_once [[noreturn]] ();

#include "hipblaslt_ostream.hpp"

#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <type_traits>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <windows.h>

#define FDOPEN(A, B) _fdopen(A, B)
#define OPEN(A) _open(A, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_APPEND, _S_IREAD | _S_IWRITE);
#define CLOSE(A) _close(A)
#else
#define FDOPEN(A, B) fdopen(A, B)
#define OPEN(A) open(A, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
#define CLOSE(A) close(A)
#endif

/***********************************************************************
 * hipblaslt_internal_ostream functions                                           *
 ***********************************************************************/

// Abort function which is called only once by hipblaslt_abort
static void hipblaslt_abort_once()
{
#ifndef _WIN32
    // Make sure the alarm and abort actions are default
    signal(SIGALRM, SIG_DFL);
    signal(SIGABRT, SIG_DFL);

    // Unblock the alarm and abort signals
    sigset_t set[1];
    sigemptyset(set);
    sigaddset(set, SIGALRM);
    sigaddset(set, SIGABRT);
    sigprocmask(SIG_UNBLOCK, set, nullptr);

    // Timeout in case of deadlock
    alarm(5);
#endif

    // Clear the map, stopping all workers
    hipblaslt_internal_ostream::clear_workers();

    // Flush all
    fflush(NULL);

    // Abort
    std::abort();
}

// Abort function which safely flushes all IO
extern "C" void hipblaslt_abort()
{
    // If multiple threads call hipblaslt_abort(), the first one wins
    static int once = (hipblaslt_abort_once(), 0);
    std::cout << once << std::endl;
}

// Get worker for writing to a file descriptor
std::shared_ptr<hipblaslt_internal_ostream::worker> hipblaslt_internal_ostream::get_worker(int fd)
{
    // For a file descriptor indicating an error, return a nullptr
    if(fd == -1)
        return nullptr;

    // C++ allows type punning of common initial sequences
    union
    {
        struct stat statbuf;
        file_id_t   file_id;
    };

    // Verify common initial sequence
    static_assert(std::is_standard_layout<file_id_t>{} && std::is_standard_layout<struct stat>{}
                      && offsetof(file_id_t, st_dev) == 0 && offsetof(struct stat, st_dev) == 0
                      && offsetof(file_id_t, st_ino) == offsetof(struct stat, st_ino)
                      && std::is_same<decltype(file_id_t::st_dev), decltype(stat::st_dev)>{}
                      && std::is_same<decltype(file_id_t::st_ino), decltype(stat::st_ino)>{},
                  "struct stat and file_id_t are not layout-compatible");

#ifndef _WIN32
    // Get the device ID and inode, to detect common files
    if(fstat(fd, &statbuf))
    {
        perror("Error executing fstat()");
        return nullptr;
    }
#else
    HANDLE                     fh = (HANDLE)_get_osfhandle(fd);
    BY_HANDLE_FILE_INFORMATION bhfi;

    if(GetFileInformationByHandle(fh, &bhfi))
    {
        // Index info should be unique
        file_id.st_dev = bhfi.nFileIndexLow;
        file_id.st_ino = bhfi.nFileIndexHigh;
    }
    else
    {
        // assign what should be unique
        file_id.st_dev = fd;
        file_id.st_ino = 0;
    }
#endif

    // Lock the map from file_id -> std::shared_ptr<hipblaslt_internal_ostream::worker>
    std::lock_guard<std::recursive_mutex> lock(worker_map_mutex());

    // Insert a nullptr map element if file_id doesn't exist in map already
    // worker_ptr is a reference to the std::shared_ptr<hipblaslt_internal_ostream::worker>
    auto& worker_ptr = worker_map().emplace(file_id, nullptr).first->second;

    // If a new entry was inserted, or an old entry is empty, create new worker
    if(!worker_ptr)
        worker_ptr = std::make_shared<worker>(fd);

    // Return the existing or new worker matching the file
    return worker_ptr;
}

// Construct hipblaslt_internal_ostream from a file descriptor
hipblaslt_internal_ostream::hipblaslt_internal_ostream(int fd)
    : m_worker_ptr(get_worker(fd))
{
    if(!m_worker_ptr)
    {
        std::cerr << "Error: Bad file descriptor " << fd << std::endl;
        hipblaslt_abort();
    }
}

// Construct hipblaslt_internal_ostream from a filename opened for writing with truncation
hipblaslt_internal_ostream::hipblaslt_internal_ostream(const char* filename)
{
    int fd       = OPEN(filename);
    m_worker_ptr = get_worker(fd);
    if(!m_worker_ptr)
    {
        std::cerr << "Cannot open " << filename << std::endl;
        hipblaslt_abort();
    }
    CLOSE(fd);
}

hipblaslt_internal_ostream::~hipblaslt_internal_ostream()
{
    flush(); // Flush any pending IO
}

// Flush the output
void hipblaslt_internal_ostream::flush()
{
    // Flush only if this stream contains a worker (i.e., is not a string)
    if(m_worker_ptr)
    {
        // The contents of the string buffer
        auto str = m_os.str();

        // Empty string buffers kill the worker thread, so they are not flushed here
        if(str.size())
            m_worker_ptr->send(std::move(str));

        // Clear the string buffer
        clear();
    }
}

void hipblaslt_internal_ostream::clear_workers()
{
    std::lock_guard<std::recursive_mutex> lock(worker_map_mutex());
    worker_map().clear();
}

// YAML Manipulators (only used for their addresses now)
std::ostream& hipblaslt_internal_ostream::yaml_on(std::ostream& os)
{
    return os;
}

std::ostream& hipblaslt_internal_ostream::yaml_off(std::ostream& os)
{
    return os;
}

/***********************************************************************
 * hipblaslt_internal_ostream::worker functions handle logging in a single thread *
 ***********************************************************************/

// Send a string to the worker thread for this stream's device/inode
// Empty strings tell the worker thread to exit
void hipblaslt_internal_ostream::worker::send(std::string str)
{
    // Create a promise to wait for the operation to complete
    std::promise<void> promise;

    // The future indicating when the operation has completed
    auto future = promise.get_future();

#ifdef _WIN32
    // Passing an empty string will make the worker thread exit.
    // The below flag will be used to handle worker thread exit condition for Windows
    bool empty_string = str.empty();
#endif

    // task_t consists of string and promise
    // std::move transfers ownership of str and promise to task
    task_t worker_task(std::move(str), std::move(promise));

    // Submit the task to the worker assigned to this device/inode
    // Hold mutex for as short as possible, to reduce contention
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(std::move(worker_task));

        // no lock needed for notification but keeping here
        m_cond.notify_one();
    }

// Wait for the task to be completed, to ensure flushed IO
#ifdef _WIN32
    if(empty_string)
        // Occassionaly this thread is not getting the promise set by the 'worker' thread during exit condition.
        // Added a timed wait to exit after one second, if we do not get the promise from worker thread.
        future.wait_for(std::chrono::seconds(1));
    else
        future.get();
#else
    future.get();
#endif
}

// Worker thread which serializes data to be written to a device/inode
void hipblaslt_internal_ostream::worker::thread_function()
{
    // Clear any errors in the FILE
    clearerr(m_file);

    // Lock the mutex in preparation for cond.wait
    std::unique_lock<std::mutex> lock(m_mutex);

    while(true)
    {
        // Wait for any data, ignoring spurious wakeups, locks lock on continue
        m_cond.wait(lock, [&] { return !m_queue.empty(); });

        // With the mutex locked, get and pop data from the front of queue
        task_t task = std::move(m_queue.front());
        m_queue.pop();

        // Temporarily unlock queue mutex, unblocking other threads
        lock.unlock();

        // An empty message indicates the closing of the stream
        if(!task.size())
        {
            // Tell future to wake up
            task.set_value();
            break;
        }

        // Write the data
        fwrite(task.data(), 1, task.size(), m_file);

        // Detect any error and flush the C FILE stream
        if(ferror(m_file) || fflush(m_file))
        {
            perror("Error writing log file");

            // Tell future to wake up
            task.set_value();
            break;
        }

        // Promise that the data has been written
        task.set_value();

        // Re-lock the mutex in preparation for cond.wait
        lock.lock();
    }
}

// Constructor creates a worker thread from a file descriptor
hipblaslt_internal_ostream::worker::worker(int fd)
{
    // The worker duplicates the file descriptor (RAII)
#ifdef _WIN32
    fd = _dup(fd);
#else
    fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
#endif

    // If the dup fails or fdopen fails, print error and abort
    if(fd == -1 || !(m_file = FDOPEN(fd, "a")))
    {
        perror("fdopen() error");
        hipblaslt_abort();
    }

    // Create a worker thread, capturing *this
    m_thread = std::thread(
#if __cplusplus >= 202002L
        [=, this]
#else
        [=]
#endif
        { thread_function(); });

    // Detach from the worker thread
    m_thread.detach();
}

hipblaslt_internal_ostream::worker::~worker()
{
    // Tell worker thread to exit, by sending it an empty string
    send({});

    // Close the FILE
    if(m_file)
        fclose(m_file);
}
