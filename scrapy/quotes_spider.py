{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import scrapy\n",
    "\n",
    "class QuotesSpider(scrapy.Spider):\n",
    "    name = \"quotes\"\n",
    "\n",
    "    def start_requests(self):\n",
    "        urls = [\n",
    "            'http://quotes.toscrape.com/page/1/',\n",
    "            'http://quotes.toscrape.com/page/2/',\n",
    "        ]\n",
    "        for url in urls:\n",
    "            yield scrapy.Request(url=url, callback=self.parse)\n",
    "\n",
    "    def parse(self, response):\n",
    "        page = response.url.split(\"/\")[-2]\n",
    "        filename = 'quotes-%s.html' % page\n",
    "        with open(filename, 'wb') as f:\n",
    "            f.write(response.body)\n",
    "        self.log('Saved file %s' % filename)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
