from features.structure_encoder import StructureEncoder

encoder = StructureEncoder()
seq = 'ACDEFGHIKL'

adj = encoder('/Users/shulei/PycharmProjects/Plaszyme/test/test_strc.pdb',seq)
print(adj)