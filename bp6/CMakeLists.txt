SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "bp6" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("bp6")
set_target_properties("bp6" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("bp6" "ceed")