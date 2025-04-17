// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

// ImGui 컨텍스트 전방 선언
struct ImGuiContext;

// 업스케일링 설정
struct UpscalingSettings {
    bool enabled = false;
    int scale = 2; // 2배 업스케일링
    float sharpness = 1.0f; // 선명도
};

struct UpscalingResources {
    VkImage upscaledImage;
    VkDeviceMemory upscaledImageMemory;
    VkImageView upscaledImageView;
    VkSampler upscaledSampler;
    
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
};

struct AllocatedBuffer {
	VkBuffer _buffer;
	VmaAllocation _allocation;
};

struct AllocatedImage {
	VkImage _image;
	VmaAllocation _allocation;
};