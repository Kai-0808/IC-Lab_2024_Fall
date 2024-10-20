import numpy as np
import struct

# Function to convert float to IEEE 754 single-precision format in hexadecimal
def float_to_ieee754_hex(f):
    # Convert float to IEEE 754 single-precision format in hexadecimal
    packed = struct.pack('!f', f)
    integer = struct.unpack('!I', packed)[0]
    return format(integer, '08X')  # 8-digit hexadecimal string

# Set random seed for reproducibility
np.random.seed(28825252)

# File paths
Img_file_path = r'../00_TESTBED/pattern_txt/Img.txt'
Ke1_file_path = r'../00_TESTBED/pattern_txt/Kernel_ch1.txt'
Ke2_file_path = r'../00_TESTBED/pattern_txt/Kernel_ch2.txt'
Wei_file_path = r'../00_TESTBED/pattern_txt/Weight.txt'
Opt_file_path = r'../00_TESTBED/pattern_txt/Opt.txt'

Out_file_path = r'../00_TESTBED/pattern_txt/Out.txt'

# Enter the number of patterns to generate
PATNUM = input("How many sets of pattern do you want to generate: ")
PATNUM = int(PATNUM)

Img_f = open(Img_file_path, 'w')
Ke1_f = open(Ke1_file_path, 'w')
Ke2_f = open(Ke2_file_path, 'w')
Wei_f = open(Wei_file_path, 'w')
Opt_f = open(Opt_file_path, 'w')

Out_f = open(Out_file_path, 'w')

def write_to_file(i, feature_map0, feature_map1, feature_map2, kernel0_ch1, kernel0_ch2, kernel1_ch1, kernel1_ch2, kernel2_ch1, kernel2_ch2, weight_ch1, weight_ch2, weight_ch3, opt):
    # Write to file
    Img_f.write(str(i) + '\n')
    for row in feature_map0:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Img_f.write(f'{ieee754}\n')
    for row in feature_map1:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Img_f.write(f'{ieee754}\n')
    for row in feature_map2:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Img_f.write(f'{ieee754}\n')
    Img_f.write('\n')

    Ke1_f.write(str(i) + '\n')
    for row in kernel0_ch1:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Ke1_f.write(f'{ieee754}\n')
    for row in kernel1_ch1:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Ke1_f.write(f'{ieee754}\n')
    for row in kernel2_ch1:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Ke1_f.write(f'{ieee754}\n')
    Ke1_f.write('\n')

    Ke2_f.write(str(i) + '\n')
    for row in kernel0_ch2:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Ke2_f.write(f'{ieee754}\n')
    for row in kernel1_ch2:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Ke2_f.write(f'{ieee754}\n')
    for row in kernel2_ch2:
        for value in row:
            ieee754 = float_to_ieee754_hex(value)
            Ke2_f.write(f'{ieee754}\n')
    Ke2_f.write('\n')

    Wei_f.write(str(i) + '\n')
    for value in weight_ch1[0]:
        ieee754 = float_to_ieee754_hex(value)
        Wei_f.write(f'{ieee754}\n')
    for value in weight_ch2[0]:
        ieee754 = float_to_ieee754_hex(value)
        Wei_f.write(f'{ieee754}\n')
    for value in weight_ch3[0]:
        ieee754 = float_to_ieee754_hex(value)
        Wei_f.write(f'{ieee754}\n')
    Wei_f.write('\n')

    Opt_f.write(str(i) + '\n')
    Opt_f.write(str(opt) + '\n')
    Opt_f.write('\n')

def write_to_output():
    Out_f.write(str(i) + '\n')
    for value in prediction:
        ieee754 = float_to_ieee754_hex(value)
        Out_f.write(f'{ieee754}\n')
    Out_f.write('\n')

def close_file():
    # Close file
    Img_f.close()
    Ke1_f.close()
    Ke2_f.close()
    Wei_f.close()
    Opt_f.close()

# Function to perform 2D convolution
def convolve2d(image, kernel):
        output = np.zeros((image.shape[0] - kernel.shape[0] + 1, 
                        image.shape[1] - kernel.shape[1] + 1))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
        return output

def convolution():
    # Calculate the output
    if(opt == 0):
        FM0_padded = np.pad(feature_map0, pad_width=1, mode='constant', constant_values=0)
        FM1_padded = np.pad(feature_map1, pad_width=1, mode='constant', constant_values=0)
        FM2_padded = np.pad(feature_map2, pad_width=1, mode='constant', constant_values=0)
    elif(opt == 1):
        FM0_padded = np.pad(feature_map0, pad_width=1, mode='edge')
        FM1_padded = np.pad(feature_map1, pad_width=1, mode='edge')
        FM2_padded = np.pad(feature_map2, pad_width=1, mode='edge')
    
    
    # Combine feature maps and kernels
    feature_maps = np.array([FM0_padded, FM1_padded, FM2_padded])
    kernels = np.array([
        [kernel0_ch1, kernel1_ch1, kernel2_ch1],
        [kernel0_ch2, kernel1_ch2, kernel2_ch2]
    ])
    # Perform convolution
    output_channels = []
    for kernel_set in kernels:
        channel_output = np.zeros((6, 6))  # Output size will be 6x6
        for f_map, kernel in zip(feature_maps, kernel_set):
            channel_output += convolve2d(f_map, kernel)
        output_channels.append(channel_output)

    # Convert to numpy array
    output = np.array(output_channels)
    return output

def max_pool2d(input_img, pool_size, stride):
    # Get dimensions
    n_channels, i_h, i_w = input_img.shape
    
    # Calculate output dimensions
    o_h = (i_h - pool_size) // stride + 1
    o_w = (i_w - pool_size) // stride + 1
    
    # Create output array
    output = np.zeros((n_channels, o_h, o_w))
    
    # Perform max pooling
    for ch in range(n_channels):
        for i in range(o_h):
            for j in range(o_w):
                output[ch, i, j] = np.max(input_img[ch, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
    return output

def activation(input_img):
    if(opt == 0):
        return 1 / (1 + np.exp(-input_img))
    elif(opt == 1):
        return np.tanh(input_img)
    
def fully_connected(input_img):
    # Flatten input
    input_flattened = input_img.flatten()
    # Initialize weights
    weights = np.array([weight_ch1, weight_ch2, weight_ch3])
    # Perform matrix multiplication
    result = np.dot(weights, input_flattened)
    return result

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def calculate_output():
    output = convolution()
    output = max_pool2d(output, 3, 3)
    output = activation(output)
    result = np.zeros((3, 1))
    result = fully_connected(output)
    result = softmax(result.flatten())
    return result

def hex_to_float(hex_string):
    """Convert a hexadecimal string to a float."""
    hex_string = ''.join(c for c in hex_string if c in '0123456789ABCDEFabcdef')
    # Pad the string to 8 characters if necessary
    hex_string = hex_string.zfill(8)
    # Convert hex to bytes
    bytes_obj = bytes.fromhex(hex_string)
    # Unpack bytes to float
    return struct.unpack('!f', bytes_obj)[0]
    
for i in range(PATNUM):

    # Generate random data
    
    feature_map0 = np.random.uniform(-0.5, 0.5, size=(5, 5))
    feature_map1 = np.random.uniform(-0.5, 0.5, size=(5, 5))
    feature_map2 = np.random.uniform(-0.5, 0.5, size=(5, 5))
    kernel0_ch1 = np.random.uniform(-0.5, 0.5, size=(2, 2))
    kernel0_ch2 = np.random.uniform(-0.5, 0.5, size=(2, 2))
    kernel1_ch1 = np.random.uniform(-0.5, 0.5, size=(2, 2))
    kernel1_ch2 = np.random.uniform(-0.5, 0.5, size=(2, 2))
    kernel2_ch1 = np.random.uniform(-0.5, 0.5, size=(2, 2))
    kernel2_ch2 = np.random.uniform(-0.5, 0.5, size=(2, 2))
    weight_ch1 = np.random.uniform(-0.5, 0.5, size=(1, 8))
    weight_ch2 = np.random.uniform(-0.5, 0.5, size=(1, 8))
    weight_ch3 = np.random.uniform(-0.5, 0.5, size=(1, 8))
    opt = np.random.randint(0, 2)
    write_to_file(i, feature_map0, feature_map1, feature_map2, kernel0_ch1, kernel0_ch2, kernel1_ch1, kernel1_ch2, kernel2_ch1, kernel2_ch2, weight_ch1, weight_ch2, weight_ch3, opt)
    prediction = np.zeros((3, 1))
    prediction = calculate_output()
    write_to_output()


# close_file()
    

    


        

