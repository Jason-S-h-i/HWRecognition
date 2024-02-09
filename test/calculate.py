
def size_cal(input, filter_size, strides, paddings):
    output = (input + 2 * paddings - filter_size )/ strides + 1
    return int(output)

conv_output1 = size_cal(448, 7, 2, 3)
maxp_output1 = size_cal(conv_output1, 2, 2, 0)
result1 = maxp_output1
conv_output2 = size_cal(result1, 3, 1, 1)
maxp_output2 = size_cal(conv_output2, 2, 2, 0)
result2 = maxp_output2
print(result1,result2)