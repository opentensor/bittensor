#!/bin/bash

python_versions=("3.8" "3.9" "3.10" "3.11")
all_passed=true

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_compatibility() {
    python_version=$1
    all_supported=0

    while read -r requirement; do
        package_name=$(echo "$requirement" | awk -F'[!=<>]' '{print $1}')
        echo -n "Checking $package_name... "

        url="https://pypi.org/pypi/$package_name/json"
        response=$(curl -s $url)
        status_code=$(curl -s -o /dev/null -w "%{http_code}" $url)

        if [ "$status_code" != "200" ]; then
            echo -e "${RED}Information not available${NC}"
            continue
        fi

        classifiers=$(echo "$response" | jq -r '.info.classifiers[]')
        requires_python=$(echo "$response" | jq -r '.info.requires_python')

        base_version="Programming Language :: Python :: ${python_version%%.*}"
        specific_version="Programming Language :: Python :: $python_version"

        if echo "$classifiers" | grep -q "$specific_version" || echo "$classifiers" | grep -q "$base_version"; then
            echo -e "${GREEN}Supported${NC}"
        elif [ "$requires_python" != "null" ] && echo "$requires_python" | grep -Eq "==$python_version|>=$python_version|<=$python_version"; then
            echo -e "${GREEN}Supported${NC}"
        else
            echo -e "${YELLOW}Warning: Specific version not listed, assuming compatibility${NC}"
        fi
    done < requirements/prod.txt

    return $all_supported
}

for version in "${python_versions[@]}"
do
    echo ""
    echo "Checking compatibility for Python $version..."
    check_compatibility $version
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}All requirements are compatible with Python $version.${NC}"
    else
        echo -e "${RED}All requirements are NOT compatible with Python $version.${NC}"
        all_passed=false
    fi
done

echo ""
if $all_passed; then
    echo -e "${GREEN}All tests passed.${NC}"
else
    echo -e "${RED}All tests did not pass.${NC}"
    exit 1
fi
