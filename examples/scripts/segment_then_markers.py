# Segment, then find markers
#
# Cluster the tissue into regions, then pull the automatic marker ions for each
# region (nearest shrunken centroids). With no groups tagged, marker/group steps
# fall back to the latest segment() clusters — so this script needs no ROIs.
#
# Run in the Script Console (Data ▸ Analysis script…) on a loaded slide.

find_peaks(snr=5)

seg = segment(n_clusters=0)          # 0 = auto-pick the cluster count by silhouette
log(f"{seg.n_clusters} regions, silhouette {seg.silhouette:.2f}")

for region_id, df in markers(top_n=10).items():
    table(df, f"Markers · region {region_id}")
