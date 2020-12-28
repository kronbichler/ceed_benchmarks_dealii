SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "benchmark_detail" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("benchmark_detail")
set_target_properties("benchmark_detail" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("benchmark_detail" "ceed")
