#!/usr/bin/env python3
import random
import os

dat_path = os.path.join(os.path.dirname(__file__), 'DRAM/dram.dat')

if __name__ == '__main__':
  with open (dat_path, 'w') as f:
    for data in range(256):
      index_a = random.randint(0, 4095)
      index_b = random.randint(0, 4095)
      index_c = random.randint(0, 4095)
      index_d = random.randint(0, 4095)
      month = random.randint(1, 12)
      if month == 2:
        day = random.randint(1, 28)
      elif month in [4, 6, 9, 11]:
        day = random.randint(1, 30)
      else:
        day = random.randint(1, 31)
      
      a_upper = (index_a >> 4) & 0xFF # 8 MSB
      a_lower = index_a & 0xF         # 4 LSB
      b_upper = (index_b >> 8) & 0x0F # 4 MSB
      b_lower = index_b & 0xFF        # 8 LSB
      
      c_upper = (index_c >> 4) & 0xFF # 8 MSB
      c_lower = index_c & 0xF         # 4 LSB
      d_upper = (index_d >> 8) & 0x0F # 4 MSB
      d_lower = index_d & 0xFF        # 8 LSB
      
      f.write(f'@{(65536 + data * 8):X}\n')
      f.write(f'{day:02X} {d_lower:02X} {c_lower:01X}{d_upper:01X} {c_upper:02X}\n')
      f.write(f'@{(65536 + data * 8 + 4):X}\n')
      f.write(f'{month:02X} {b_lower:02X} {a_lower:01X}{b_upper:01X} {a_upper:02X}\n')
      
      if data != 255:
        f.write(f'\n')
