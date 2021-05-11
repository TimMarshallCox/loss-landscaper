import numpy as np

import matplotlib.pyplot as plt

loss_matrix = np.load('loss_matrix.npy')
xs = np.load('xs.npy')
ys = np.load('ys.npy')
resolution = np.load('resolution.npy')
pcs = np.load('pcs.npy')


logged_loss_matrix = np.load('logged_loss_matrix.npy')
test_loss = np.load('test_loss.npy')

plt.title('Logged Loss Landscape')
plt.contourf(xs, ys, logged_loss_matrix - 3)
plt.xticks(xs[0::resolution])
plt.yticks(ys[0::resolution])
plt.colorbar()
# plt.plot(trained_network[0][0], trained_network[0][1], c='white', marker='*')
# colour = test_loss[1:11]
colour = np.linspace(1, 0, len(pcs[0]), endpoint=False)
plt.scatter(pcs[0], pcs[1], c=colour, cmap='YlOrRd', marker='.')
# plt.scatter(pcs[0], pcs[1], c='white', marker='.')
plt.show()


train_loss = np.load('train_loss.npy')
test_loss = np.load('test_loss.npy')

plt.plot(np.linspace(0, 10, 940), train_loss)
plt.plot(range(0, 11), test_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Convergence Plot')
plt.grid()
plt.show()

proportion = np.load('proportion.npy')
cumulative_proportion = np.load('cumulative_proportion.npy')

plt.plot(range(1, 11), proportion)
plt.plot(range(1, 11), cumulative_proportion)
plt.legend(['Proportion Explained', 'Cumulative Proportion Explained'])
plt.xticks(range(1, 11))
plt.xlabel('Principal Components')
plt.ylabel('Proportion Explained')
plt.title('Scree Plot')
plt.grid()
plt.show()
