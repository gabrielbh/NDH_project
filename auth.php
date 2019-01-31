require 'vendor/autoload.php';

$session = new SpotifyWebAPI\Session(
    '87158cc41aa945e0b7be77d94a57efa8',
    'fd77db134a3a45b7b3f1fa85f43375b9'
);

$session->requestCredentialsToken();
$accessToken = $session->getAccessToken();

// Store the access token somewhere. In a database for example.

// Send the user along and fetch some data!
header('Location: app.php');
die();