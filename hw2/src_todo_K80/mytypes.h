#ifndef MYTYPES_H
#define MYTYPES_H


#define BLOCKTILE_M 96 
#define BLOCKTILE_N 64
#define BLOCKTILE_K 32
#define MC BLOCKTILE_M
#define KC BLOCKTILE_K
#define NC BLOCKTILE_N

#define TM (BLOCKTILE_M/BLOCKDIM_Y)
#define TN (BLOCKTILE_N/BLOCKDIM_X)

#define TW 16

#endif
