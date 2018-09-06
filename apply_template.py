# -*- coding: utf-8 -*-
import json

if __name__ == '__main__':
    with open('README.template', 'r', encoding="utf-8") as file:
        text = file.readlines()
    text = ''.join(text)

    with open('results.json', 'r', encoding="utf-8") as file:
        results = json.load(file)

    for i in range(20):
        text = text.replace('$(result_{})'.format(i), '{}, prob: {}'.format(results[i]['label'], results[i]['prob']))

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)
