// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#ifndef CK_CODE_GEN_RTC

#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <map>

namespace ck {
namespace internal {
template <typename T>
struct ParseEnvVal
{
};

template <>
struct ParseEnvVal<bool>
{
    static bool parse_env_var_value(const char* vp)
    {
        std::string value_env_str{vp};

        for(auto& c : value_env_str)
        {
            if(std::isalpha(c) != 0)
            {
                c = std::tolower(static_cast<unsigned char>(c));
            }
        }

        if(value_env_str == "disable" || value_env_str == "disabled" || value_env_str == "0" ||
           value_env_str == "no" || value_env_str == "off" || value_env_str == "false")
        {
            return false;
        }
        else if(value_env_str == "enable" || value_env_str == "enabled" || value_env_str == "1" ||
                value_env_str == "yes" || value_env_str == "on" || value_env_str == "true")
        {
            return true;
        }
        else
        {
            throw std::runtime_error("Invalid value for env variable");
        }

        return false; // shouldn't reach here
    }
};

// Supports hexadecimals (with leading "0x"), octals (if prefix is "0") and decimals (default).
// Returns 0 if environment variable is in wrong format (strtoull fails to parse the string).
template <>
struct ParseEnvVal<uint64_t>
{
    static uint64_t parse_env_var_value(const char* vp) { return std::strtoull(vp, nullptr, 0); }
};

template <>
struct ParseEnvVal<std::string>
{
    static std::string parse_env_var_value(const char* vp) { return std::string{vp}; }
};

template <typename T>
struct EnvVar
{
    private:
    T value{};
    bool is_unset = true;

    public:
    const T& GetValue() const { return value; }

    bool IsUnset() const { return is_unset; }

    void Unset() { is_unset = true; }

    void UpdateValue(const T& val)
    {
        is_unset = false;
        value    = val;
    }

    explicit EnvVar(const char* const name, const T& def_val)
    {
        // NOLINTNEXTLINE (concurrency-mt-unsafe)
        const char* vp = std::getenv(name);
        if(vp != nullptr) // a value was provided
        {
            is_unset = false;
            value    = ParseEnvVal<T>::parse_env_var_value(vp);
        }
        else // no value provided, use default value
        {
            value = def_val;
        }
    }
};
} // end namespace internal

// static inside function hides the variable and provides
// thread-safety/locking
// Used in global namespace
#define CK_DECLARE_ENV_VAR(name, type, default_val)                            \
    namespace ck::env {                                                        \
    struct name                                                                \
    {                                                                          \
        static_assert(std::is_same_v<name, ::ck::env::name>,                   \
                      "CK_DECLARE_ENV* must be used in the global namespace"); \
        using value_type = type;                                               \
        static ck::internal::EnvVar<type>& Ref()                               \
        {                                                                      \
            static ck::internal::EnvVar<type> var{#name, default_val};         \
            return var;                                                        \
        }                                                                      \
    };                                                                         \
    }

#define CK_DECLARE_ENV_VAR_BOOL(name) CK_DECLARE_ENV_VAR(name, bool, false)

#define CK_DECLARE_ENV_VAR_UINT64(name) CK_DECLARE_ENV_VAR(name, uint64_t, 0)

#define CK_DECLARE_ENV_VAR_STR(name) CK_DECLARE_ENV_VAR(name, std::string, "")

#define CK_ENV(name) \
    ck::env::name {}

template <class EnvVar>
inline const std::string& EnvGetString(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, std::string>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool EnvIsEnabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    return !EnvVar::Ref().IsUnset() && EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool EnvIsDisabled(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, bool>);
    return !EnvVar::Ref().IsUnset() && !EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline uint64_t EnvValue(EnvVar)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, uint64_t>);
    return EnvVar::Ref().GetValue();
}

template <class EnvVar>
inline bool EnvIsUnset(EnvVar)
{
    return EnvVar::Ref().IsUnset();
}

template <class EnvVar>
void EnvUnset(EnvVar)
{
    EnvVar::Ref().Unset();
}

/// updates the cached value of an environment variable
template <typename EnvVar, typename ValueType>
void UpdateEnvVar(EnvVar, const ValueType& val)
{
    static_assert(std::is_same_v<typename EnvVar::value_type, ValueType>);
    EnvVar::Ref().UpdateValue(val);
}

template <typename EnvVar>
void UpdateEnvVar(EnvVar, const std::string_view& val)
{
    EnvVar::Ref().UpdateValue(
        ck::internal::ParseEnvVal<typename EnvVar::value_type>::parse_env_var_value(val.data()));
}

} // namespace ck

// environment variable to enable logging:
// export CK_LOGGING=ON or CK_LOGGING=1 or CK_LOGGING=ENABLED
CK_DECLARE_ENV_VAR_BOOL(CK_LOGGING)

#endif
