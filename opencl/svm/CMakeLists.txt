project(svm)

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
add_executable(svm svm.cpp )
target_link_libraries(svm OpenCL)

add_test(NAME svm_test 
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
		 COMMAND svm)


