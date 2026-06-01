const puppeteer = require('puppeteer');
const buildTraceFileName = process.env.BUILD_TRACE_FILE;

(async () => {
	try {
		// Launch the browser
		const browser = await puppeteer.launch({
			args: [
				'--no-sandbox',
				'--headless',
				'--disable-gpu',
				'--window-size=1920x1080'
		]});
		const page = await browser.newPage();
		await page.setViewport({ width: 1920, height: 1080 });
		await page.goto('https://ui.perfetto.dev');
		// Wait for the home page to be visible
		console.log('Waiting for page to load...');
		await page.waitForSelector('.pf-home-page', { visible: true, timeout: 30000 });
		// Locate and click the Open trace button
		const elements = await page.$$('li');
		let element = null;
		for (const el of elements) {
			const text = await el.evaluate(node => node.textContent);
			if (text && text.includes('Open trace file')) {
				element = el;
				break;
			}
		}
		if (element) {
			const [fileChooser] = await Promise.all([
				page.waitForFileChooser(),
				element.click()
			]);
			await fileChooser.accept([`/workspace/${buildTraceFileName}`]);
		} else {
			throw new Error('Element not found');
		}
		console.log('Waiting for data to load...');
		// Wait for the timeline element to be visible
		await page.waitForSelector('.pf-track', { timeout: 30000 });
		// Wait for the data to finish loading
		await page.waitForFunction(() => {
			return !document.body.textContent.includes('Loading...');
		}, { timeout: 30000 });
		console.log('Capturing screenshot...');
		await page.screenshot({path: '/workspace/perfetto_snapshot_build.png'});
		console.log('Done capturing screenshot...');
		await browser.close();
	} catch (err) {
		console.error(err);
		process.exit(1);
	}
})();