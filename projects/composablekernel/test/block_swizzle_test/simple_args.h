#pragma once

#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <assert.h>

struct arg_content_t
{
    std::string name; // key
    std::string value;
    std::string help_text;
};

class simple_args_t
{
    public:
    simple_args_t() {}
    simple_args_t& insert(const std::string& name_,
                          const std::string& default_value_,
                          const std::string& help_text_)
    {
        arg_content_t arg{name_, default_value_, help_text_};

        if(arg_map.count(arg.name) != 0)
        {
            std::cout << "arg:" << arg.name << "already exist" << std::endl;
        }
        else
        {
            arg_map[arg.name] = arg;
        }
        return *this;
    }
    void usage()
    {
        for(auto& content : arg_map)
        {
            std::vector<std::string> help_text_lines;
            size_t pos = 0;
            for(size_t next_pos = content.second.help_text.find('\n', pos);
                next_pos != std::string::npos;)
            {
                help_text_lines.push_back(
                    std::string(content.second.help_text.begin() + pos,
                                content.second.help_text.begin() + next_pos++));
                pos      = next_pos;
                next_pos = content.second.help_text.find('\n', pos);
            }
            help_text_lines.push_back(std::string(content.second.help_text.begin() + pos,
                                                  content.second.help_text.end()));

            int arg_name_width = 16 - content.second.name.length();
            arg_name_width     = arg_name_width > 0 ? arg_name_width : 2;
            std::cout << std::setw(4) << "-" << content.second.name << std::setw(arg_name_width)
                      << " " << help_text_lines[0] << std::endl;

            for(auto help_next_line = std::next(help_text_lines.begin());
                help_next_line != help_text_lines.end();
                ++help_next_line)
            {
                std::cout << std::setw(28) << " " << *help_next_line << std::endl;
            }
        }
    }
    bool parse(int argc, char* argv[], int start_index = 1)
    {
        if(argc <= start_index)
        {
            // std::cout << "not enough args (" << argc << ") with starting index " << start_index
            // << std::endl;
            return true;
        }
        for(int i = start_index; i < argc; i++)
        {
            std::string cur_arg = std::string(argv[i]);
            if(cur_arg[0] != '-')
            {
                std::cout << "illegal input" << std::endl;
                usage();
                return false;
            }
            else if(cur_arg[0] == '-' && cur_arg[1] == '?')
            {
                usage();
                return false;
            }
            else
            {
                size_t found_equal = cur_arg.find('=');
                if(found_equal == std::string::npos || found_equal == (cur_arg.length() - 1))
                {
                    std::cout << "failed while parsing \"" << cur_arg << "\", "
                              << "arg must be in the form \"-name=value\"" << std::endl;
                    return false;
                }
                std::string arg_name  = cur_arg.substr(1, found_equal - 1);
                std::string arg_value = cur_arg.substr(found_equal + 1);
                if(arg_map.count(arg_name) == 0)
                {
                    std::cout << "no such arg \"" << arg_name << "\" registered" << std::endl;
                    return false;
                }
                arg_map[arg_name].value = arg_value;
            }
        }
        return true;
    }

    std::string get(const std::string& name) const { return get_str(name); }

    std::string get_str(const std::string& name) const
    {
        assert(arg_map.count(name) != 0);
        std::string value = arg_map.at(name).value;
        return value;
    }

    int get_int(const std::string& name) const
    {
        assert(arg_map.count(name) != 0);
        int value = atoi(arg_map.at(name).value.c_str());
        return value;
    }

    uint32_t get_uint32(const std::string& name) const
    {
        assert(arg_map.count(name) != 0);
        uint32_t value = strtoul(arg_map.at(name).value.c_str(), nullptr, 10);
        return value;
    }

    uint64_t get_uint64(const std::string& name) const
    {
        assert(arg_map.count(name) != 0);
        uint64_t value = strtoull(arg_map.at(name).value.c_str(), nullptr, 10);
        return value;
    }

    double get_double(const std::string& name) const
    {
        assert(arg_map.count(name) != 0);
        double value = atof(arg_map.at(name).value.c_str());
        return value;
    }

    float get_float(const std::string& name) const
    {
        assert(arg_map.count(name) != 0);
        float value = atof(arg_map.at(name).value.c_str());
        return value;
    }

    private:
    std::unordered_map<std::string, arg_content_t> arg_map;
};
