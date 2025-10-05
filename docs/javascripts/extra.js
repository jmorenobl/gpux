/* GPUX Documentation Custom JavaScript */

// Add custom functionality here
document.addEventListener('DOMContentLoaded', function() {
  console.log('GPUX Documentation loaded');

  // Add copy code button functionality enhancement
  const codeBlocks = document.querySelectorAll('pre code');

  codeBlocks.forEach(function(block) {
    // Add line numbers if needed
    // Add syntax highlighting enhancements
  });

  // Analytics for external links
  const externalLinks = document.querySelectorAll('a[href^="http"]');

  externalLinks.forEach(function(link) {
    link.addEventListener('click', function(e) {
      // Track external link clicks
      const href = this.getAttribute('href');
      console.log('External link clicked:', href);
    });
  });

  // Smooth scroll to anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });
});
