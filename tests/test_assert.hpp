#pragma once
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#define TEST_ASSERT(cond)                                                                                              \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::cerr << "Assertion failed: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n";               \
            throw std::runtime_error("Test failed");                                                                   \
        }                                                                                                              \
    } while (0)
