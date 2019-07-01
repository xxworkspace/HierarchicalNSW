
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

  // ΢��: �ض�ֵ
  justedMaxAbsoluteFloatValue = maxAbsoluteFloatValue * 127 / 128;
  // ��------ ����: ����λ & float �ľ���ֵ ------��

  // ����: ������ -> λ����ռ�
  memcpy(&bitSpace, &floatValue, 4);

  // ����: ����λ
  sign = bitSpace & 0x80000000;
  bitSpace = bitSpace & 0x7fffffff;

  // ��ȡ: �������ľ���ֵ <- λ����ռ�
  memcpy(&absoluteFloatValue, &bitSpace, 4);

  // ��------ ����: float ��ָ��λ  ------��

  // --- �ض�: float �ľ���ֵ ---
  if (absoluteFloatValue > justedMaxAbsoluteFloatValue){
    absoluteFloatValue = justedMaxAbsoluteFloatValue;
  }

  // ����: �������ľ���ֵ + maxAbsoluteFloatValue ��ֵ
  absoluteFloatValue = absoluteFloatValue + maxAbsoluteFloatValue;
 
  // ��------ ��ȡ: ��Ч���� ------��

  // ����: �������ľ���ֵ -> λ����ռ�
  memcpy(&bitSpace, &absoluteFloatValue, 4);

  // ����: �Ƿ��λ
  roundNumber = bitSpace >> 15;
  roundNumber = roundNumber & 1;

  // ��ȡ: �������ľ���ֵ��ǰ 7 λ��Ч����
  intValue = bitSpace >> 16;
  intValue = intValue & 0x7f;

  // ��λ
  int8Value = intValue + roundNumber;

  // ��------ �ϲ�: ����λ & ������ ------��

  // --- ���: ����λ -> int8 �ľ���ֵ ---
  if (sign == 0x80000000){
    int8Value = -int8Value;
  }
  // ����: ת���õ� int8
  return int8Value;
}

void floats2int8(char* target,float* src,int num,float max){
  for(int i = 0 ;i < num ; i++)
  target[i] = float2int8(src[i],max);
}
