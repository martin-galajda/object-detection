import csv

def get_domains_and_urls():
    domains = []
    urls = []
    with open('./scraper/foto-domains-2019-03.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')

        row_idx = 0
        for row in csvreader:
            if row_idx == 0:
                row_idx += 1
                continue
            domain, country = row[:2]
            domains += [f'{domain}.{country}']
            urls += [f'http://{domain}.{country}']
            row_idx += 1
    print(domains)
    print(urls)

    return {
        'domains': domains,
        'urls': urls,
    }

