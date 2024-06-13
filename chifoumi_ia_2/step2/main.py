import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

rock_dir = os.path.join('rps/rock')
paper_dir = os.path.join('rps/paper')
scissors_dir = os.path.join('rps/scissors')

rock_files = os.listdir(rock_dir)

paper_files = os.listdir(paper_dir)

scissors_files = os.listdir(scissors_dir)

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) for fname in scissors_files[pic_index-2:pic_index]]

# try to plot the images
for img_path in next_rock:
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

for img_path in next_paper:
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

for img_path in next_scissors:
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()