# Header stamped at query time
std_msgs/Header header

# Query and candidate map timestamps in microseconds
int64 query_time
int64 candidate_time

# Sequential IDs in the online buffer
int32 query_index
int32 candidate_index

# Best match score and self-similarity gap
float64 score
float64 min_dist

# Saved image paths for query and candidate
string query_image_path
string candidate_image_path

# Local map resolution
float64 resolution
