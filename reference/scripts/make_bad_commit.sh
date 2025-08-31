#!/bin/bash
# Script to create a bad commit for bisect testing
# Changes only pad direction or truncate length in summarization

set -e

echo "Creating bad commit for bisect testing..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Create a backup of the current summarization training script
cp reference/tasks/summarization/train.py reference/tasks/summarization/train.py.backup

# Make the change - flip pad direction from right to left
echo "Modifying summarization training script..."
sed -i 's/pad_direction: str = "right"/pad_direction: str = "left"/' reference/tasks/summarization/train.py

# Also change the truncate_at default to create more divergence
sed -i 's/truncate_at: Optional\[int\] = None/truncate_at: Optional\[int\] = 256/' reference/tasks/summarization/train.py

# Stage the changes
git add reference/tasks/summarization/train.py

# Create the bad commit
git commit -m "BAD COMMIT: Changed pad direction to left and truncate_at to 256 for bisect testing

This commit intentionally introduces changes that will cause divergence:
- Changed pad_direction from 'right' to 'left'
- Changed truncate_at from None to 256

This is used for testing the bisect functionality."

echo "Bad commit created successfully!"
echo "Commit hash: $(git rev-parse HEAD)"
echo ""
echo "To revert this change:"
echo "  git revert HEAD"
echo "  # or"
echo "  cp reference/tasks/summarization/train.py.backup reference/tasks/summarization/train.py"
echo ""
echo "To continue with bisect:"
echo "  git bisect bad"