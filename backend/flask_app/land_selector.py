import pickle
import numpy as np
from scipy.spatial import distance_matrix


class LandSelector:
    def __init__(self, length_labels, length_grids):
        self.length_labels = length_labels
        self.length_grids = length_grids
        self.farming_area_labels = [0, 2, 5, 6]
        self.cost_matrix = np.zeros((length_grids, length_grids))
        self.cost_coefficients = np.array([1, 1, 1, -1, -1, 1, 1, -1, 1, 1])
        self.counts = np.zeros(10)

    def get_label_encoder(self, pickle_file):
        # sklearn preprocessing LabelEncoder class is returned.
        label_encoder = pickle.load(open(pickle_file, 'rb'))
        # get original labels
        labels = label_encoder.inverse_transform(range(self.length_labels))
        return labels

    def organize_prediction_data(self, labels):
        # data: 39x39 numpy ndarray
        #  each element -> 64 x 64 pixels field where each pixel 10 meters IRL.
        # labels: 10 labels, numpy ndarray
        # get the count of coordinates for the label with highest number of coordinates
        unique, counts = np.unique(labels, return_counts=True)
        for i, lbl in enumerate(unique):
            self.counts[lbl] = counts[i]
        self.counts = self.counts.astype(np.int64)
        max_num_label = np.max(self.counts).astype(np.int64)
        # placeholder (-1, -1) filled array with shape (#labels, #max_elements_label)
        grid_coordinates = -1 * np.ones((self.length_labels, max_num_label))
        # (#labels, #max_elements_label) -> (#labels, #max_elements_label, 2)
        grouped_data = np.stack((grid_coordinates, grid_coordinates), axis=2)
        valid_grid_coordinates = np.zeros((self.length_labels, max_num_label))
        del grid_coordinates
        for i in range(self.length_labels):
            # get #elements a label has
            current_label_count = self.counts[i]
            if current_label_count == 0:
                continue
            # filter the indices (i, j) -> [0, 0]...[38, 38]
            mask = [labels == i]
            filtered_labels = np.where(mask)
            del mask
            current_label_indices = np.stack((filtered_labels[1], filtered_labels[2]), axis=1)
            del filtered_labels
            # write the indices to the data array
            grouped_data[i, :current_label_count, :] = current_label_indices
            valid_grid_coordinates[i, :current_label_count] = np.ones((1, current_label_count))
            del current_label_indices
        return grouped_data, valid_grid_coordinates

    def distance_calculator(self):
        pass

    def cost_function(self, grouped_data):
        # grouped_data -> (#labels, #max_elements_label, 2), numpy array
        x, y, _ = grouped_data.shape
        for i in self.farming_area_labels:
            for j in range(x):
                if i == j or self.counts[i] == 0 or self.counts[j] == 0:
                    continue
                ith_label_data = grouped_data[i, :self.counts[i], :]
                jth_label_data = grouped_data[j, :self.counts[j], :]
                dist_matrix = distance_matrix(ith_label_data, jth_label_data)
                # ---------------
                # | x  x  x  I  |
                # | x  f  I  x  |
                # | S  x  x  x  |
                # ---------------
                dist_matrix = 1 / dist_matrix
                dist_matrix = np.sum(dist_matrix, axis=1)
                self.cost_matrix[np.array(ith_label_data[:, 0], dtype=int), np.array(ith_label_data[:, 1], dtype=int)] \
                    += self.cost_coefficients[j] * dist_matrix

    def calculate_land_scores(self):
        pass

    def get_land_scores(self, score_threshold=35):
        grid_dict = {}
        for i in range(self.length_grids):
            for j in range(self.length_grids):
                grid_dict["{}_{}".format(i, j)] = self.cost_matrix[i, j]
        grid_dict_sorted = {k: v for k, v in sorted(grid_dict.items(), key=lambda item: item[1], reverse=True)
                            if v > score_threshold}
        del grid_dict
        return grid_dict_sorted


# land_selector = LandSelector(length_labels=10, length_grids=39)
# grid_labels = land_selector.get_label_encoder('./le.pkl')
# print(grid_labels)
# trial_labels = np.random.randint(10, size=(39, 39))

# grouped_data, valid_grid_coordinates = land_selector.organize_prediction_data(trial_labels)
# land_selector.cost_function(grouped_data)
# grid_dict = land_selector.get_land_scores()

# np.savetxt('cost_matrix.txt', cost_matrix, fmt='%1.4e')

