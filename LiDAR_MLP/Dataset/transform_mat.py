from glob import glob


path = '/home/mikep/DataSets/KITTI/Images/Train/calibration'

fnames = glob(path+'/*.txt')

transforms = []

for name in fnames:
    fobj = open(name, 'r')

    lines = []
    for line in fobj:
        lines.append(line.split())

    lines = lines[:7]

    P0 = [float(element) for element in lines[0][1:]]
    P1 = [float(element) for element in lines[1][1:]]
    P2 = [float(element) for element in lines[2][1:]]
    P3 = [float(element) for element in lines[3][1:]]
    R0_rect = [float(element) for element in lines[4][1:]]
    Tr_velo_to_cam = [float(element) for element in lines[5][1:]]
    Tr_imu_to_velo = [float(element) for element in lines[6][1:]]

    entry = {"P0": P0, "P1": P1, "P2": P2, "P3":P3, "R0_rect": R0_rect, "Tr_velo_to_cam": Tr_velo_to_cam, "Tr_imu_to_velo": Tr_imu_to_velo}
    transforms.append(entry)

sum_P0 = [0] * len(P0)
sum_P1 = [0] * len(P1)
sum_P2 = [0] * len(P2)
sum_P3 = [0] * len(P3)
sum_R0_rect = [0] * len(R0_rect)
sum_Tr_velo_to_cam = [0] * len(Tr_velo_to_cam)
sum_Tr_imu_to_velo = [0] * len(Tr_imu_to_velo)

for entry in transforms:
    P0 = entry['P0']
    P1 = entry['P1']
    P2 = entry['P2']
    P3 = entry['P3']
    R0_rect = entry['R0_rect']
    Tr_velo_to_cam = entry['Tr_velo_to_cam']
    Tr_imu_to_velo = entry['Tr_imu_to_velo']

    for i in range(len(P0)): sum_P0[i] += P0[i]
    for i in range(len(P1)): sum_P1[i] += P1[i]
    for i in range(len(P2)): sum_P2[i] += P2[i]
    for i in range(len(P3)): sum_P3[i] += P3[i]
    for i in range(len(R0_rect)): sum_R0_rect[i] += R0_rect[i]
    for i in range(len(Tr_velo_to_cam)): sum_Tr_velo_to_cam[i] += Tr_velo_to_cam[i]
    for i in range(len(Tr_imu_to_velo)): sum_Tr_imu_to_velo[i] += Tr_imu_to_velo[i]

avg_P0 = [elem/len(transforms) for elem in sum_P0]
avg_P1 = [elem/len(transforms) for elem in sum_P1]
avg_P2 = [elem/len(transforms) for elem in sum_P2]
avg_P3 = [elem/len(transforms) for elem in sum_P3]
avg_R0_rect = [elem/len(transforms) for elem in sum_R0_rect]
avg_Tr_velo_to_cam = [elem/len(transforms) for elem in sum_Tr_velo_to_cam]
avg_Tr_imu_to_velo = [elem/len(transforms) for elem in sum_Tr_imu_to_velo]

output = open('avg_transform.txt', 'w')

write_P0 = 'P0: '
for elem in avg_P0:
    write_P0 += str(elem) + ' '
write_P0 += '\n'

write_P1 = 'P1: '
for elem in avg_P1:
    write_P1 += str(elem) + ' '
write_P1 += '\n'

write_P2 = 'P2: '
for elem in avg_P2:
    write_P2 += str(elem) + ' '
write_P2 += '\n'

write_P3 = 'P3: '
for elem in avg_P3:
    write_P3 += str(elem) + ' '
write_P3 += '\n'

write_R0_rect = 'R0_rect: '
for elem in avg_R0_rect:
    write_R0_rect += str(elem) + ' '
write_R0_rect += '\n'

write_Tr_velo_to_cam = 'Tr_velo_to_cam: '
for elem in avg_Tr_velo_to_cam:
    write_Tr_velo_to_cam += str(elem) + ' '
write_Tr_velo_to_cam += '\n'

write_Tr_imu_to_velo = 'Tr_imu_to_velo: '
for elem in avg_Tr_imu_to_velo:
    write_Tr_imu_to_velo += str(elem) + ' '

output.write(write_P0)
output.write(write_P1)
output.write(write_P2)
output.write(write_P3)
output.write(write_R0_rect)
output.write(write_Tr_velo_to_cam)
output.write(write_Tr_imu_to_velo)

output.close()
