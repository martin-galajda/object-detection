import json
import tldextract
import csv

from collections import defaultdict
dict_res_keys = ['image_count_per_domain_counter', 'websites_visited_per_domain_counter']

def write_csv(out, header, data_rows):
    with open(out, 'w') as out_file:
        out_file.write(header)
        out_file.write('\n')
        for row in data_rows:
            output_str = ''
            for cell in row:
                output_str += f'{cell},'
            output_str = output_str.strip(',')
            output_str += '\n'
            out_file.write(output_str)


def load_photo_domains_dict(filepath  = './scraper/foto-domains-2019-03.csv'):
    res_dict = defaultdict(int)
    with open(filepath, 'r') as csv_photo_domains:
        csvreader = csv.reader(csv_photo_domains, delimiter=',')

        row_idx = 0
        for row in csvreader:
            if row_idx == 0:
                row_idx += 1
                continue
            domain, page_views = [row[0], row[2]]
            res_dict[domain] = page_views
            row_idx += 1

    return res_dict


def process_results(res_dict):
    domain_total_count_pairs = []
    domain_avg_count_pairs = []
    domain_websites_crawled_count_pairs = []
    domain_all_counts_pairs = []

    image_count_results = res_dict[dict_res_keys[0]]
    websites_crawled_count_results = res_dict[dict_res_keys[1]]

    image_count_by_domain = defaultdict(int)
    avg_website_count_by_domain = defaultdict(int)
    websites_crawled_count_by_domain = defaultdict(int)

    photo_domains_dict = load_photo_domains_dict()

    for key in image_count_results.keys():
        ext = tldextract.extract(key)
        image_count_by_domain[ext.domain] += image_count_results[key]

    for domain in image_count_by_domain.keys():
        if photo_domains_dict[domain] != 0:
            domain_total_count_pairs += [(domain, image_count_by_domain[domain])]

    for key in websites_crawled_count_results.keys():
        ext = tldextract.extract(key)
        websites_crawled_count_by_domain[ext.domain] += websites_crawled_count_results[key]

    for domain in websites_crawled_count_by_domain.keys():
        if photo_domains_dict[domain] != 0:
            domain_websites_crawled_count_pairs += [(domain, websites_crawled_count_by_domain[domain])]
            avg_website_count_by_domain[domain] = int(image_count_by_domain[domain] / websites_crawled_count_by_domain[domain])

    for domain in avg_website_count_by_domain.keys():
        if photo_domains_dict[domain] != 0:
            domain_avg_count_pairs += [(domain, avg_website_count_by_domain[domain])]


    for domain in avg_website_count_by_domain.keys():
        if photo_domains_dict[domain] != 0:
            domain_all_counts_pairs += [(
                domain,
                avg_website_count_by_domain[domain],
                image_count_by_domain[domain],
                websites_crawled_count_by_domain[domain],
                photo_domains_dict[domain]
            )]

    domain_total_count_pairs = sorted(domain_total_count_pairs, key = lambda x: -x[1])
    domain_avg_count_pairs = sorted(domain_avg_count_pairs, key = lambda x: -x[1])
    domain_websites_crawled_count_pairs = sorted(domain_websites_crawled_count_pairs, key = lambda x: -x[1])
    domain_all_counts_pairs = sorted(domain_all_counts_pairs, key = lambda x: -x[1])


    return {
        'domain_total_count_pairs': domain_total_count_pairs,
        'domain_avg_count_pairs': domain_avg_count_pairs,
        'domain_websites_crawled_count_pairs': domain_websites_crawled_count_pairs,
        'domain_all_counts_pairs': domain_all_counts_pairs,
    }

with open('./scraper/out.json', 'r') as file:
    res_dict = json.load(file)

    results = process_results(res_dict)

    write_csv('./scraper/total_image_count_per_domain.csv', 'domain, total_image_count', results['domain_total_count_pairs'])
    write_csv('./scraper/avg_image_count_per_domain_website.csv', 'domain, avg_image_count', results['domain_avg_count_pairs'])
    write_csv('./scraper/total_websites_crawled_per_domain.csv', 'domain, websites_crawled_count', results['domain_websites_crawled_count_pairs'])
    write_csv('./scraper/domain_all_counts_pairs.csv', 'domain, '
                                                       'avg_image_count, '
                                                       'total_image_count, '
                                                       'websites_crawled_count, '
                                                       'page_views_15_min_window', results['domain_all_counts_pairs'])
