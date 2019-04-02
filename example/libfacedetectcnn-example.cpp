/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2019, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetectcnn.h"

// define the buffer size. Do not change the size!
#define DETECT_BUFFER_SIZE 0x20000

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <image_file_name> <output_file_name>\n", argv[0]);
    return -1;
  }

  // load an image and convert it to gray (single-channel)
  cv::Mat image = cv::imread(argv[1]);
  if (image.empty()) {
    fprintf(stderr, "Can not load the image file %s.\n", argv[1]);
    return -1;
  }

  // input_buffer is used in the detection functions.
  // If you call functions in multiple threads, please create one buffer for each thread!
  unsigned char *input_buffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
  if (!input_buffer) {
    fprintf(stderr, "Can not alloc buffer.\n");
    return -1;
  }

  ///////////////////////////////////////////
  // CNN face detection
  // Best detection rate
  //////////////////////////////////////////
  //!!! The input image must be a BGR one (three-channel) instead of RGB
  //!!! DO NOT RELEASE output_buffer !!!
  int *output_buffer = facedetect_cnn(input_buffer, (unsigned char *)(image.ptr(0)), image.cols, image.rows, (int)image.step);
  printf("{\"count\": %d, \n\"faces\": [\n", (output_buffer ? *output_buffer : 0));

  cv::Mat result_cnn = image.clone();
  // print the detection results
  for (int i = 0; i < (output_buffer ? *output_buffer : 0); i++) {
    short *p = ((short *)(output_buffer + 1)) + 142 * i;
    int x = p[0], y = p[1], w = p[2], h = p[3], confidence = p[4], angle = p[5];
    printf("{\"position\": [%d, %d, %d, %d], \"confidence\": %d, \"angle\": %d}\n", x, y, w, h, confidence, angle);
    cv::rectangle(result_cnn, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
  }
  printf("]}");

  cv::imwrite(argv[2], result_cnn);

  // imshow("result_cnn", result_cnn);
  // waitKey();

  // release the input buffer
  free(input_buffer);

  return 0;
}
