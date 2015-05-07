project(dp)

cmake_minimum_required(VERSION 2.8)

enable_testing()

set(bitness "x86")
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(bitness "x86_64")
endif()

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

include_directories($ENV{AMDAPPSDKROOT}/include)
link_directories($ENV{AMDAPPSDKROOT}/lib/${bitness})

#add_library()
add_executable(dp dp.cpp )
target_link_libraries(dp OpenCL)

add_test(NAME dp_test 
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		 COMMAND dp)


