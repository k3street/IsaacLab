from pxr import Usd, UsdGeom

USD_PATH = "../openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/props/Collected_SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01.usd"

stage = Usd.Stage.Open(USD_PATH)
prim = stage.GetDefaultPrim()
print("Default prim:", prim.GetPath(), prim.GetTypeName())

bbox_cache = UsdGeom.BBoxCache(0.0, ["default"])
bbox = bbox_cache.ComputeWorldBound(prim)
aligned = bbox.ComputeAlignedBox()
print("Aligned bbox min:", aligned.GetMin())
print("Aligned bbox max:", aligned.GetMax())
print("Size:", aligned.GetMax() - aligned.GetMin())
