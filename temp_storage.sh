# Create temp space in home
mkdir -p /home/csimmon2/sym_pt_temp

# For each session, symlink the loc directory
for path in \
  sub-007/ses-03/derivatives/fsl/loc \
  sub-007/ses-04/derivatives/fsl/loc \
  sub-017/ses-04/derivatives/fsl/loc; do
    
    src="/user_data/csimmon2/sym_pt/$path"
    tmp="/home/csimmon2/sym_pt_temp/$(echo $path | tr '/' '_')"
    
    # Move existing content to temp
    mkdir -p "$tmp"
    cp -r "$src"/* "$tmp"/ 2>/dev/null
    
    # Replace with symlink
    rm -rf "$src"
    ln -s "$tmp" "$src"
    
    echo "Linked: $src -> $tmp"
done

'''
# Move back later
for path in ...; do
    src="/user_data/csimmon2/sym_pt/$path"
    tmp=$(readlink "$src")
    rm "$src"
    mv "$tmp" "$src"
done
'''