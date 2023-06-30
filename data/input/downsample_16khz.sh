mkdir -p audios_16khz
for i in audios/*.wav; do
    o=audios_16khz/${i#audios/}
    sox "$i" -r 16000 "${o%}"
done