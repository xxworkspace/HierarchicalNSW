
#include "int8.h"
#include <memory.h>

inline char float2int8(float floatValue, float maxAbsoluteFloatValue)
{
  // C++
  char int8Value = 0;

  int sign = 0;
  int intValue = 0;
  int roundNumber = 0;

  int bitSpace = 0;

  float absoluteFloatValue = 0;
  float justedMaxAbsoluteFloatValue = 0;
  
  floatValue /= maxAbsoluteFloatValue;
  maxAbsoluteFloatValue = 1.0;

  // 微调: 截断值
  justedMaxAbsoluteFloatValue = maxAbsoluteFloatValue * 127 / 128;
  // ↓------ 分离: 符号位 & float 的绝对值 ------↓

  // 复制: 浮点数 -> 位运算空间
  memcpy(&bitSpace, &floatValue, 4);

  // 分离: 符号位
  sign = bitSpace & 0x80000000;
  bitSpace = bitSpace & 0x7fffffff;

  // 获取: 浮点数的绝对值 <- 位运算空间
  memcpy(&absoluteFloatValue, &bitSpace, 4);

  // ↓------ 对齐: float 的指数位  ------↓

  // --- 截断: float 的绝对值 ---
  if (absoluteFloatValue > justedMaxAbsoluteFloatValue){
    absoluteFloatValue = justedMaxAbsoluteFloatValue;
  }

  // 计算: 浮点数的绝对值 + maxAbsoluteFloatValue 的值
  absoluteFloatValue = absoluteFloatValue + maxAbsoluteFloatValue;
 
  // ↓------ 获取: 有效数字 ------↓

  // 复制: 浮点数的绝对值 -> 位运算空间
  memcpy(&bitSpace, &absoluteFloatValue, 4);

  // 计算: 是否进位
  roundNumber = bitSpace >> 15;
  roundNumber = roundNumber & 1;

  // 获取: 浮点数的绝对值的前 7 位有效数字
  intValue = bitSpace >> 16;
  intValue = intValue & 0x7f;

  // 进位
  int8Value = intValue + roundNumber;

  // ↓------ 合并: 符号位 & 运算结果 ------↓

  // --- 添加: 符号位 -> int8 的绝对值 ---
  if (sign == 0x80000000){
    int8Value = -int8Value;
  }
  // 返回: 转换好的 int8
  return int8Value;
}

void floats2int8(char* target,float* src,int num,float max){
  for(int i = 0 ;i < num ; i++)
  target[i] = float2int8(src[i],max);
}
