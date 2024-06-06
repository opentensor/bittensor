#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide a Python version as an argument."
    exit 1
fi

python_version="$1"
all_passed=true

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_compatibility() {
    all_supported=0

    while read -r requirement; do
        # Skip lines starting with git+
        if [[ "$requirement" == git+* ]]; then
            continue
        fi

        package_name=$(echo "$requirement" | awk -F'[!=<>~]' '{print $1}' | awk -F'[' '{print $1}') # Strip off brackets
        echo -n "Checking $package_name... "

        url="https://pypi.org/pypi/$package_name/json"
        response=$(curl -s $url)
        status_code=$(curl -s -o /dev/null -w "%{http_code}" $url)

        if [ "$status_code" != "200" ]; then
            echo -e "${RED}Information not available for $package_name. Failure.${NC}"
            all_supported=1
            continue
        fi

        classifiers=$(echo "$response" | jq -r '.info.classifiers[]')
        requires_python=$(echo "$response" | jq -r '.info.requires_python')

        base_version="Programming Language :: Python :: ${python_version%%.*}"
        specific_version="Programming Language :: Python :: $python_version"

        if echo "$classifiers" | grep -q "$specific_version" || echo "$classifiers" | grep -q "$base_version"; then
            echo -e "${GREEN}Supported${NC}"
        elif [ "$requires_python" != "null" ]; then
            if echo "$requires_python" | grep -Eq "==$python_version|>=$python_version|<=$python_version"; then
                echo -e "${GREEN}Supported${NC}"
            else
                echo -e "${RED}Not compatible with Python $python_version due to constraint $requires_python.${NC}"
                all_supported=1
            fi
        else
            echo -e "${YELLOW}Warning: Specific version not listed, assuming compatibility${NC}"
        fi
    done < requirements/prod.txt

    return $all_supported
}

echo "Checking compatibility for Python $python_version..."
check_compatibility
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All requirements are compatible with Python $python_version.${NC}"
else
    echo -e "${RED}All requirements are NOT compatible with Python $python_version.${NC}"
    all_passed=false
fi

echo ""
if $all_passed; then
    echo -e "${GREEN}All tests passed.${NC}"
else
    echo -e "${RED}All tests did not pass.${NC}"
    exit 1
fi
