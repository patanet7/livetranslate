#define WIN32_LEAN_AND_MEAN
#include <winsock2.h> // Include this first to avoid conflicts
#include <windows.h>

#include <iostream>
#include <boost/system/error_code.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

int main() {
    // Test Boost.System
    boost::system::error_code ec;
    std::cout << "Boost.System test - error category: " << ec.category().name() << std::endl;

    // Test Boost.Date_Time
    boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();
    std::cout << "Boost.Date_Time test - current time: " << now << std::endl;

    return 0;
} 