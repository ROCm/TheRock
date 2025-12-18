// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Test Logging Wrapper
 * ====================
 * C++ wrapper around TheRock logging framework for test applications
 */

#ifndef THEROCK_TEST_LOGGING_HPP
#define THEROCK_TEST_LOGGING_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <ctime>

/**
 * Simple logger class for C++ tests
 * Mimics Python logging framework behavior
 */
class TestLogger {
public:
    enum class Level {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
        CRITICAL = 4
    };

private:
    std::string component_;
    Level level_;
    
    // ANSI color codes
    static constexpr const char* COLOR_RESET = "\033[0m";
    static constexpr const char* COLOR_CYAN = "\033[36m";
    static constexpr const char* COLOR_GREEN = "\033[32m";
    static constexpr const char* COLOR_YELLOW = "\033[33m";
    static constexpr const char* COLOR_RED = "\033[31m";
    static constexpr const char* COLOR_MAGENTA = "\033[35m";
    
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
    
    const char* get_level_name(Level lvl) const {
        switch (lvl) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO: return "INFO";
            case Level::WARNING: return "WARNING";
            case Level::ERROR: return "ERROR";
            case Level::CRITICAL: return "CRITICAL";
            default: return "UNKNOWN";
        }
    }
    
    const char* get_level_color(Level lvl) const {
        switch (lvl) {
            case Level::DEBUG: return COLOR_CYAN;
            case Level::INFO: return COLOR_GREEN;
            case Level::WARNING: return COLOR_YELLOW;
            case Level::ERROR: return COLOR_RED;
            case Level::CRITICAL: return COLOR_MAGENTA;
            default: return COLOR_RESET;
        }
    }
    
    void log_message(Level lvl, const std::string& message) const {
        if (lvl < level_) return;
        
        const char* color = get_level_color(lvl);
        const char* level_name = get_level_name(lvl);
        
        std::cout << get_timestamp() << " - "
                  << component_ << " - "
                  << color << level_name << COLOR_RESET << " - "
                  << message << std::endl;
    }

public:
    TestLogger(const std::string& component = "test", Level level = Level::DEBUG)
        : component_(component), level_(level) {}
    
    void set_level(Level level) { level_ = level; }
    Level get_level() const { return level_; }
    
    // Basic logging methods
    void debug(const std::string& message) const {
        log_message(Level::DEBUG, message);
    }
    
    void info(const std::string& message) const {
        log_message(Level::INFO, message);
    }
    
    void warning(const std::string& message) const {
        log_message(Level::WARNING, message);
    }
    
    void error(const std::string& message) const {
        log_message(Level::ERROR, message);
    }
    
    void critical(const std::string& message) const {
        log_message(Level::CRITICAL, message);
    }
    
    // Template methods for formatted logging (C++11 compatible)
    template<typename... Args>
    void debug(const std::string& format, Args... args) const {
        log_message(Level::DEBUG, format_string(format, args...));
    }
    
    template<typename... Args>
    void info(const std::string& format, Args... args) const {
        log_message(Level::INFO, format_string(format, args...));
    }
    
    template<typename... Args>
    void warning(const std::string& format, Args... args) const {
        log_message(Level::WARNING, format_string(format, args...));
    }
    
    template<typename... Args>
    void error(const std::string& format, Args... args) const {
        log_message(Level::ERROR, format_string(format, args...));
    }
    
    template<typename... Args>
    void critical(const std::string& format, Args... args) const {
        log_message(Level::CRITICAL, format_string(format, args...));
    }

private:
    // Simple format string implementation (Python-style {} replacement)
    template<typename T>
    std::string format_string_impl(const std::string& format, size_t& pos, T value) const {
        std::stringstream ss;
        size_t start = pos;
        pos = format.find("{}", start);
        
        if (pos != std::string::npos) {
            ss << format.substr(start, pos - start);
            ss << value;
            pos += 2;
        } else {
            ss << format.substr(start);
        }
        
        return ss.str();
    }
    
    template<typename T, typename... Args>
    std::string format_string_impl(const std::string& format, size_t& pos, T value, Args... args) const {
        std::stringstream ss;
        ss << format_string_impl(format, pos, value);
        std::string remaining = format.substr(pos);
        size_t new_pos = 0;
        ss << format_string_impl(remaining, new_pos, args...);
        return ss.str();
    }
    
    template<typename... Args>
    std::string format_string(const std::string& format, Args... args) const {
        size_t pos = 0;
        return format_string_impl(format, pos, args...);
    }
    
    // Base case for recursion
    std::string format_string(const std::string& format) const {
        return format;
    }
};

/**
 * RAII timer for logging operation duration
 */
class ScopedTimer {
private:
    TestLogger logger_;
    std::string operation_;
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    ScopedTimer(const TestLogger& logger, const std::string& operation)
        : logger_(logger), operation_(operation),
          start_(std::chrono::high_resolution_clock::now()) {
        logger_.debug("Starting operation: " + operation_);
    }
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        logger_.info("✅ Completed operation: " + operation_ + 
                    " (" + std::to_string(duration.count()) + "ms)");
    }
};

// Helper macro for timed operations
#define TIMED_OPERATION(logger, operation_name) \
    ScopedTimer _timer_##__LINE__(logger, operation_name)

#endif // THEROCK_TEST_LOGGING_HPP

