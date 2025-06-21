//
// Created by linus on 19.03.24.
//
#pragma once
#include <string>

class OptionalTensorBoardLogger
{
   public:
    OptionalTensorBoardLogger(std::string path);
    void add_scalar(std::string name, int epoch_id, float value);
};