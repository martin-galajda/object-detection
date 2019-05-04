import json
from scraper.collect_annotable_urls.crawl_configuration import CrawlConfiguration
from typing import List
from urllib.parse import urlparse


def load_configuration():
    def load_json_config(path_to_file = './scraper/collect_annotable_urls/config.json'):
        with open(path_to_file, 'r') as json_file_config:
            config = json.load(json_file_config)

            return config

    def get_root_domains_for_crawling(config_file: dict):
        domains = list(map(lambda website_config: urlparse(website_config['url']).netloc, config_file['domains']))

        return domains

    def get_start_urls_for_crawling(config_file: dict):
        start_urls = list(map(lambda website_config: website_config['url'], config_file['domains']))

        return start_urls

    def get_crawl_configs(config_file: dict) -> List[CrawlConfiguration]:
        configs = list(map(lambda website_config: CrawlConfiguration(website_config), config_file['domains']))

        return configs

    json_config = load_json_config()

    return {
        'root_domains': get_root_domains_for_crawling(json_config),
        'crawl_configs': get_crawl_configs(json_config),
        'start_urls': get_start_urls_for_crawling(json_config)
    }
