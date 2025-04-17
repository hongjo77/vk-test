#include <vk_engine.h>
#include <iostream>

int main(int argc, char* argv[]) {
    VulkanEngine engine;
    
    try {
        engine.init();
        engine.run();
        engine.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
