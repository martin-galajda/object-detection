from urllib.parse import urlparse


class CrawlConfiguration(object):
    url: str = None
    domain: str = None

    pageWithGalleryContainsAnchorHrefRegexp: str = None
    pageWithGalleryContainsImgSrcRegexp: str = None
    pageWithGalleryUrlMatchesRegexp: str = None
    pageWithGalleryUrlHasToMatchRegexp: str = None

    galleryUrlMatchesRegexp: str = None

    def __init__(self, json_config):
        self.url = json_config['url']
        self.domain = urlparse(self.url).netloc

        self.pageWithGalleryContainsAnchorHrefRegexp = json_config['pageWithGalleryContainsAnchorHrefRegexp']
        self.pageWithGalleryContainsImgSrcRegexp = json_config['pageWithGalleryContainsImgSrcRegexp']
        self.pageWithGalleryUrlMatchesRegexp = json_config['pageWithGalleryUrlMatchesRegexp']
        self.galleryUrlMatchesRegexp = json_config['galleryUrlMatchesRegexp']

        if 'pageWithGalleryUrlHasToMatchRegexp' in json_config:
            self.pageWithGalleryUrlHasToMatchRegexp = json_config['pageWithGalleryUrlHasToMatchRegexp']
