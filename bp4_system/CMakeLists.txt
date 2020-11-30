SET(TARGET_SRC bench.cc)

ADD_EXECUTABLE( "bp4_system" ${TARGET_SRC})
DEAL_II_SETUP_TARGET("bp4_system")
set_target_properties("bp4_system" PROPERTIES OUTPUT_NAME "bench")
TARGET_LINK_LIBRARIES("bp4_system" "ceed")
