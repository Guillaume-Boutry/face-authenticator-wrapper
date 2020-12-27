// FROM https://github.com/Kagami/go-face under CC0 license
#ifndef LOAD_MEM_JPEG_H
#define LOAD_MEM_JPEG_H

typedef struct {
    dlib::matrix<dlib::rgb_pixel> img;
} Image;

Image Load_mem_jpeg(const uint8_t* img_data, int len);
#endif //LOAD_MEM_JPEG_H
