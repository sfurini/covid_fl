#!/bin/bash

# convert plink formats to text files
plink --bfile /path_to_M1/finalMask/MergeM1 --recode --tab --out /path_to_M1/finalMask/MergeM1

exit
