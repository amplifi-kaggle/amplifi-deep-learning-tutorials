import random

def computeLongestPalindrome(text):
    # BEGIN_YOUR_CODE
    return ''
    # END_YOUR_CODE
    
print 1, computeLongestPalindrome("")
print 2, computeLongestPalindrome("ab")
print 3, computeLongestPalindrome("aa")
print 4, computeLongestPalindrome("animal")

import random
numChars = 5
length = 400
random.seed(42)
text = ' '.join(chr(random.randint(ord('a'), ord('a') + numChars - 1)) for _ in range(length))
print 'text =', text[:50]
print 5, computeLongestPalindrome(text)