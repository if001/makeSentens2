#! /bin/sh

files=`ls | grep .*txt`
echo "to_utf8.sh"
for file in $files
do
    echo $file
    nkf -w $file > "utf8_"$file
done

exit 0

