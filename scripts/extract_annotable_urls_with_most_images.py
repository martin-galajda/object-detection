import argparse
import csv
from collections import defaultdict


def sort_urls_by_images_count(urls_by_domain):
    sorted_urls_by_images_for_each_domain = {}

    for domain in urls_by_domain:
        domain_urls = urls_by_domain[domain]
        sorted_urls_by_images_for_each_domain[domain] = list(sorted(domain_urls, key=lambda url: url["number_of_images_found"], reverse=True))

    return sorted_urls_by_images_for_each_domain


def pick_top_urls_by_number_of_images_from_each_domain(urls_by_domain, number_of_urls_to_pick = 250):
    sorted_urls_by_images_for_each_domain = sort_urls_by_images_count(urls_by_domain)

    curr_ptr_to_urls_list_by_domain = defaultdict(lambda: 0)

    domains = list(sorted_urls_by_images_for_each_domain.keys())
    domains_length = len(domains)

    out_list = []

    for _ in range(number_of_urls_to_pick):
        for curr_domain_ptr in range(domains_length):
            domain = domains[curr_domain_ptr]
            curr_ptr_for_domain = curr_ptr_to_urls_list_by_domain[domain]

            if curr_ptr_for_domain < len(sorted_urls_by_images_for_each_domain[domain]):
                out_list += [sorted_urls_by_images_for_each_domain[domain][curr_ptr_for_domain]["url_data"]]
                curr_ptr_to_urls_list_by_domain[domain] += 1

    return out_list


def main(arguments):
    path_to_input_csv_file = arguments.path_to_collected_annotation_csv_file
    path_to_output_csv_file = arguments.out_path

    urls_by_domain = defaultdict(list)

    csv_dict_keys = None

    with open(path_to_input_csv_file, 'r') as fp:
        csv_reader = csv.DictReader(fp)
        line_idx = 0
        for row in csv_reader:
            line_idx += 1

            number_of_images_found = int(row["total_img_elements_found"])

            urls_by_domain[row["domain"]] += [{
                "number_of_images_found": number_of_images_found,
                "url_data": row
            }]

            csv_dict_keys = row.keys()

    out_list = pick_top_urls_by_number_of_images_from_each_domain(urls_by_domain)

    with open(path_to_output_csv_file, 'w') as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_dict_keys)
        writer.writeheader()

        for url_data in out_list:
            writer.writerow(url_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_collected_annotation_csv_file',
                        type=str,
                        default='./scraper/out/collect_annotable_urls/output-2019-07-10T09:43:16-export.csv',
                        required=False,
                        help='Path to CSV file containing annotable URLs of webpages with photogalleries.')

    parser.add_argument('--out_path',
                        type=str,
                        default='./scraper/out/collect_annotable_urls/output-2019-07-10T09:43:16-export-top-5000.csv',
                        required=False,
                        help='Path to output file containing 5000 annotable urls.')
    args = parser.parse_args()

    main(args)