/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "auxiliary.hpp"

#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <hipblaslt/hipblaslt.h>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <queue>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <utility>
#ifdef _WIN32
#include <io.h>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#define STDOUT_FILENO _fileno(stdout)
#define STDERR_FILENO _fileno(stderr)
#define FDOPEN(A, B) _fdopen(A, B)
#define OPEN(A) _open(A, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_APPEND, _S_IREAD | _S_IWRITE);
#define CLOSE(A) _close(A)
#else
#define FDOPEN(A, B) fdopen(A, B)
#define OPEN(A) open(A, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
#define CLOSE(A) close(A)
#include <unistd.h>
#endif

extern "C" HIPBLASLT_EXPORT void hipblaslt_abort() __attribute__((__noreturn__));

/*****************************************************************************
 * hipblaslt output streams                                                    *
 *****************************************************************************/

#define hipblaslt_cout (hipblaslt_internal_ostream::cout())
#define hipblaslt_cerr (hipblaslt_internal_ostream::cerr())

/***************************************************************************
 * The hipblaslt_internal_ostream class performs atomic IO on log files, and provides *
 * consistent formatting                                                   *
 ***************************************************************************/

class HIPBLASLT_EXPORT hipblaslt_internal_ostream
{
    /**************************************************************************
     * The worker class sets up a worker thread for writing to log files. Two *
     * files are considered the same if they have the same device ID / inode. *
     **************************************************************************/
    class worker
    {
        // task_t represents a payload of data and a promise to finish
        class task_t
        {
            std::string        m_str;
            std::promise<void> m_promise;

        public:
            // The task takes ownership of the string payload and promise
            task_t(std::string&& str, std::promise<void>&& promise)
                : m_str(std::move(str))
                , m_promise(std::move(promise))
            {
            }

            // Notify the future to wake up
            void set_value()
            {
                m_promise.set_value();
            }

            // Size of the string payload
            size_t size() const
            {
                return m_str.size();
            }

            // Data of the string payload
            const char* data() const
            {
                return m_str.data();
            }
        };

        // FILE is used for safety in the presence of signals
        FILE* m_file = nullptr;

        // This worker's thread
        std::thread m_thread;

        // Condition variable for worker notification
        std::condition_variable m_cond;

        // Mutex for this thread's queue
        std::mutex m_mutex;

        // Queue of tasks
        std::queue<task_t> m_queue;

        // Worker thread which waits for and handles tasks sequentially
        void thread_function();

    public:
        // Worker constructor creates a worker thread for a raw filehandle
        explicit worker(int fd);

        // Send a string to be written
        void send(std::string);

        // Destroy a worker when all std::shared_ptr references to it are gone
        ~worker();
    };

    // Two filehandles point to the same file if they share the same (std_dev, std_ino).

    // Initial slice of struct stat which contains device ID and inode
    struct file_id_t
    {
        dev_t st_dev; // ID of device containing file
        ino_t st_ino; // Inode number
    };

    // Compares device IDs and inodes for map containers
    struct file_id_less
    {
        bool operator()(const file_id_t& lhs, const file_id_t& rhs) const
        {
            return lhs.st_ino < rhs.st_ino || (lhs.st_ino == rhs.st_ino && lhs.st_dev < rhs.st_dev);
        }
    };

    // Map from file_id to a worker shared_ptr
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& worker_map()
    {
        static std::map<file_id_t, std::shared_ptr<worker>, file_id_less> file_id_to_worker_map;
        return file_id_to_worker_map;
    }

    // Mutex for accessing the map
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& worker_map_mutex()
    {
        static std::recursive_mutex map_mutex;
        return map_mutex;
    }

    // Output buffer for formatted IO
    std::ostringstream m_os;

    // Worker thread for accepting tasks
    std::shared_ptr<worker> m_worker_ptr;

    // Flag indicating whether YAML mode is turned on
    bool m_yaml = false;

    // Get worker for file descriptor
    static std::shared_ptr<worker> get_worker(int fd);

    // Private explicit copy constructor duplicates the worker and starts a new buffer
    explicit hipblaslt_internal_ostream(const hipblaslt_internal_ostream& other)
        : m_worker_ptr(other.m_worker_ptr)
    {
    }

public:
    // Default constructor is a std::ostringstream with no worker
    hipblaslt_internal_ostream() = default;

    // Move constructor
    hipblaslt_internal_ostream(hipblaslt_internal_ostream&&) = default;

    // Move assignment
    hipblaslt_internal_ostream& operator=(hipblaslt_internal_ostream&&) & = default;

    // Copy assignment is deleted
    hipblaslt_internal_ostream& operator=(const hipblaslt_internal_ostream&) = delete;

    // Construct from a file descriptor, which is duped
    explicit hipblaslt_internal_ostream(int fd);

    // Construct from a C filename
    explicit hipblaslt_internal_ostream(const char* filename);

    // Construct from a std::string filename
    explicit hipblaslt_internal_ostream(const std::string& filename)
        : hipblaslt_internal_ostream(filename.c_str())
    {
    }

    // Create a duplicate of this
    hipblaslt_internal_ostream dup() const
    {
        if(!m_worker_ptr)
            throw std::runtime_error("Attempting to duplicate a hipblaslt_internal_ostream "
                                     "without an associated file");
        return hipblaslt_internal_ostream(*this);
    }

    // For testing to allow file closing and deletion
    static void clear_workers();

    // Convert stream output to string
    std::string str() const
    {
        return m_os.str();
    }

    // Clear the buffer
    void clear()
    {
        m_os.clear();
        m_os.str({});
    }

    // Flush the output
    void flush();

    // Destroy the hipblaslt_internal_ostream
    virtual ~hipblaslt_internal_ostream();

    // Implemented as singleton to avoid the static initialization order fiasco
    static hipblaslt_internal_ostream& cout()
    {
        thread_local hipblaslt_internal_ostream t_cout{STDOUT_FILENO};
        return t_cout;
    }

    // Implemented as singleton to avoid the static initialization order fiasco
    static hipblaslt_internal_ostream& cerr()
    {
        thread_local hipblaslt_internal_ostream t_cerr{STDERR_FILENO};
        return t_cerr;
    }

    // Abort function which safely flushes all IO
    friend void hipblaslt_abort_once();

    /*************************************************************************
     * Non-member friend functions for formatted output                      *
     *************************************************************************/

    // Default output for non-enumeration types
    template <typename T, std::enable_if_t<!std::is_enum<std::decay_t<T>>{}, int> = 0>
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, T&& x)
    {
        os.m_os << std::forward<T>(x);
        return os;
    }

    // Default output for enumeration types
    template <typename T, std::enable_if_t<std::is_enum<std::decay_t<T>>{}, int> = 0>
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, T&& x)
    {
        os.m_os << std::underlying_type_t<std::decay_t<T>>(x);
        return os;
    }

    // Pairs for YAML output
    template <typename T1, typename T2>
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  std::pair<T1, T2>           p)
    {
        os << p.first << ": ";
        os.m_yaml = true;
        os << p.second;
        os.m_yaml = false;
        return os;
    }

    // Floating-point output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, double x)
    {
        if(!os.m_yaml)
            os.m_os << x;
        else
        {
            // For YAML, we must output the floating-point value exactly
            if(std::isnan(x))
                os.m_os << ".nan";
            else if(std::isinf(x))
                os.m_os << (x < 0 ? "-.inf" : ".inf");
            else
            {
                char s[32];
                snprintf(s, sizeof(s) - 2, "%.17g", x);

                // If no decimal point or exponent, append .0 to indicate floating point
                for(char* end = s; *end != '.' && *end != 'e' && *end != 'E'; ++end)
                {
                    if(!*end)
                    {
                        end[0] = '.';
                        end[1] = '0';
                        end[2] = '\0';
                        break;
                    }
                }
                os.m_os << s;
            }
        }
        return os;
    }

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  hipblasLtHalf               half)
    {
        return os << float(half);
    }

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, hip_bfloat16 bf16)
    {
        return os << float(bf16);
    }

    // Integer output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, int32_t x)
    {
        os.m_os << x;
        return os;
    }

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, uint32_t x)
    {
        os.m_os << x;
        return os;
    }

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, int64_t x)
    {
        os.m_os << x;
        return os;
    }

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, uint64_t x)
    {
        os.m_os << x;
        return os;
    }

    // bool output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, bool b)
    {
        if(os.m_yaml)
            os.m_os << (b ? "true" : "false");
        else
            os.m_os << (b ? 1 : 0);
        return os;
    }

    // Character output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, char c)
    {
        if(os.m_yaml)
        {
            char s[]{c, 0};
            os.m_os << std::quoted(s, '\'');
        }
        else
            os.m_os << c;
        return os;
    }

    // String output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, const char* s)
    {
        if(os.m_yaml)
            os.m_os << std::quoted(s);
        else
            os.m_os << s;
        return os;
    }

    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  const std::string&          s)
    {
        return os << s.c_str();
    }

    // hipDataType output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os, hipDataType d)
    {
        os.m_os << hip_datatype_to_string(d);
        return os;
    }

    // hipblasComputeType_t output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  hipblasComputeType_t        d)
    {
        os.m_os << hipblas_computetype_to_string(d);
        return os;
    }

    // hipblasOperation_t output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  hipblasOperation_t          trans)

    {
        return os << hipblas_operation_to_string(trans);
    }

    // hipblasStatus_t output
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  hipblasStatus_t             status)
    {
        os.m_os << hipblas_status_to_string(status);
        return os;
    }

    // Transfer hipblaslt_internal_ostream to std::ostream
    friend std::ostream& operator<<(std::ostream& os, const hipblaslt_internal_ostream& str)
    {
        return os << str.str();
    }

    // Transfer hipblaslt_internal_ostream to hipblaslt_internal_ostream
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream&       os,
                                                  const hipblaslt_internal_ostream& str)
    {
        return os << str.str();
    }

    // IO Manipulators
    friend hipblaslt_internal_ostream& operator<<(hipblaslt_internal_ostream& os,
                                                  std::ostream& (*pf)(std::ostream&))
    {
        // Turn YAML formatting on or off
        if(pf == hipblaslt_internal_ostream::yaml_on)
            os.m_yaml = true;
        else if(pf == hipblaslt_internal_ostream::yaml_off)
            os.m_yaml = false;
        else
        {
            // Output the manipulator to the buffer
            os.m_os << pf;

            // If the manipulator is std::endl or std::flush, flush the output
            if(pf == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)
               || pf == static_cast<std::ostream& (*)(std::ostream&)>(std::flush))
            {
                os.flush();
            }
        }
        return os;
    }

    // YAML Manipulators (only used for their addresses now)
    static std::ostream& yaml_on(std::ostream& os);
    static std::ostream& yaml_off(std::ostream& os);
};
