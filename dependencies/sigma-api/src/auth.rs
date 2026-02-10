//! JWT authentication middleware.

use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: usize,
    pub scopes: Vec<String>,
}

pub struct AuthLayer {
    secret: String,
}

impl AuthLayer {
    pub fn new(secret: &str) -> Self {
        AuthLayer {
            secret: secret.to_string(),
        }
    }

    pub fn generate_token(&self, user_id: &str, scopes: Vec<String>) -> Result<String, String> {
        let claims = Claims {
            sub: user_id.to_string(),
            exp: (chrono_epoch_secs() + 3600) as usize,
            scopes,
        };
        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.secret.as_bytes()),
        )
        .map_err(|e| e.to_string())
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, String> {
        decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.secret.as_bytes()),
            &Validation::default(),
        )
        .map(|data| data.claims)
        .map_err(|e| e.to_string())
    }
}

fn chrono_epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_and_validate_token() {
        let auth = AuthLayer::new("test-secret-key-for-unit-tests-only");
        let token = auth.generate_token("user-1", vec!["infer".into()]).unwrap();
        let claims = auth.validate_token(&token).unwrap();
        assert_eq!(claims.sub, "user-1");
        assert!(claims.scopes.contains(&"infer".to_string()));
    }

    #[test]
    fn test_invalid_token() {
        let auth = AuthLayer::new("secret");
        assert!(auth.validate_token("garbage.token.here").is_err());
    }
}
