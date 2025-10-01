if(NOT DEFINED RL_TOOLS_ROOT)
  message(FATAL_ERROR "RL_TOOLS_ROOT is not defined")
endif()

include(px4_add_library)

px4_add_module(
	MODULE modules__mc_raptor
	MAIN mc_raptor
	STACK_MAIN 2000
	COMPILE_FLAGS
		-DARM_MATH_LOOPUNROLL
	SRCS
		RLtoolsPolicy.cpp
		rl_tools_adapter.cpp
	# DEPENDS
	# 	px4_work_queue
	EXTERNAL
	)
target_include_directories(
	modules__mc_raptor
	PRIVATE
	${RL_TOOLS_ROOT}/include
	${RL_TOOLS_ROOT}/data
)

target_compile_features(modules__mc_raptor PRIVATE cxx_std_17)
target_compile_options(modules__mc_raptor PRIVATE -Wno-error -DARM_MATH_LOOPUNROLL)




