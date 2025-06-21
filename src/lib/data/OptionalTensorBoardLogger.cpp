//
// Created by linus on 19.03.24.
//
#include "OptionalTensorBoardLogger.h"

// compiled in based on OS
#ifdef TBLOGGER
#    include "tensorboard_logger.h"
std::shared_ptr<TensorBoardLogger> tblogger;

#endif

OptionalTensorBoardLogger::OptionalTensorBoardLogger(std::string path)
{
#ifdef TBLOGGER

    tblogger = std::make_shared<TensorBoardLogger>(path.c_str());
#endif
}


void OptionalTensorBoardLogger::add_scalar(std::string name, int epoch_id, float value)
{
#ifdef TBLOGGER

    tblogger->add_scalar(name, epoch_id, value);
#endif
}