#!/bin/bash

for md_file in md_*.xyz; do
    if [[ ! -f "$md_file" ]]; then
        echo "No MD files found matching pattern md_*.xyz"
        exit 0
    fi

    temp_string=$(basename "$md_file" .xyz | sed 's/md_//')
    dir_name="DATA_EXP_${temp_string}"
    echo "Processing $md_file into $dir_name..."

    mkdir -p "$dir_name"
    cp "$md_file" xyz_to_raw_obs.py raw_to_set.sh "$dir_name/"

    cd "$dir_name" || exit 1
    python xyz_to_raw_obs.py --input_traj "$(basename "$md_file")"
    bash raw_to_set.sh
    cd ..
done

echo "All workflows completed successfully!"
