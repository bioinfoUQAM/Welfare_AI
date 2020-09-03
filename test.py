from Welfare_AI_dataset import CowsDataset
import pandas as pd
import os


DICTIONARY_PATH = os.path.join(os.path.dirname(__file__), 'Dictionary_Kinematics.xlsx')
dictionary = pd.read_excel(DICTIONARY_PATH, "Video File -> Excel Tab Names")
dictionary = dictionary.to_numpy()
names = CowsDataset.get_side_sheets('side2')
dataset = CowsDataset(names)
# print(len(dataset))
# for i, x in enumerate(dataset):
#     print("i: ", i)
#     print("x: ", x)
# print(CowsDataset)
columns = CowsDataset.get_joint_names(dataset)
data = dataset[0]['Z-E']
# print('column', columns)
#list_of_joints = CowsDataset.get_list_of_joints(dataset)
#Hierarchical_clustering.cosine_correlation(list_of_joints, 3)
#Hierarchical_clustering.plot_corr_trees(dataset)
#GeneratedData.generate_samples(dataset, 0, 'Y-E')
#GeneratedData.create_generated_cow_data(dataset, 0)
# GeneratedData.generate_monotone_sequences(data)
# name = CowsDataset.get_cow_name(names[0])


#Plot_dataset.plot_subplots_joint(dataset)


