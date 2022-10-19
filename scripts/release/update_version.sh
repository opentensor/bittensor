#!/bin/bash

VERSION=$(cat VERSION)
CODE_WITH_VERSION='bittensor/__init__.py'

MAJOR=$(awk -F. '{print $1}' <<< $VERSION)
MINOR=$(awk -F. '{print $2}' <<< $VERSION)
PATCH=$(awk -F. '{print $3}' <<< $VERSION)

# RC version
RC=$(awk -F- '{print $NF}' <<< $version)
if [ -z $RC ]; then
    echo "Current version: $MAJOR.$MINOR.$PATCH"
else
    echo "Current version: $MAJOR.$MINOR.$PATCH-$RC"
fi

OPERATION=$1
case $OPERATION in
    "major")
        echo "Applying a $OPERATION update"
        NEW_VERSION="$((MAJOR + 1)).$MINOR.$PATCH"
        ;;
    "minor")
        echo "Applying a $OPERATION update"
        NEW_VERSION="$MAJOR.$((MINOR + 1)).$PATCH"
        ;;
    "patch")
        echo "Applying a $OPERATION update"
        NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
        ;;
    "rc")
        SUFFIX=$2
        if [ -z $SUFFIX ]; then
            echo "Suffix is needed when updating version to a RC"
            exit 1
        fi
        NEW_VERSION="$MAJOR.$MINOR.$PATCH-$SUFFIX"
        ;;
    *)
    echo "This operation is not allowed. Try one of the following: {major, minor, patch, rc}"
    exit 1
    ;;
esac

echo "New version: $NEW_VERSION"

#sed -i "18,22s/$VERSION/$NEW_VERSION/g" $CODE_WITH_VERSION
#echo -n $NEW_VERSION > VERSION