from typing import List
import httpx
from fastapi import HTTPException, Depends
from .config import settings
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/token')


def validate_remotely(token, issuer, client_id, client_secret):
    headers = {
        'accept': 'application/json',
        'cache-control': 'no-cache',
        'content-type': 'application/x-www-form-urlencoded',
    }
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'token': token,
    }
    url = issuer + '/introspect'

    response = httpx.post(url, headers=headers, data=data)
    return response
    # return response.status_code == httpx.codes.OK and response.json()['active']


# use keycloak's introspection endpoint to validate token
# Depends(oauth2_scheme) means that request should have 'Authorization' header containing a "Bearer token" value.
# If so token will be returned in str, otherwise 401 will be returned.
def validate(token: str = Depends(oauth2_scheme), ) -> List[str]:
    res = validate_remotely(
        token=token,
        issuer=settings.token_issuer,
        client_id=settings.client_id,
        client_secret=settings.client_secret,
    )
    if res.status_code == httpx.codes.OK and res.json()['active']:
        return res.json().get('realm_access', {}).get('roles')
    else:
        raise HTTPException(status_code=401)


class RemoteAuthValidator:
    def __init__(self, allowed_roles: List):
        self.allowed_roles = allowed_roles

    # if user role exits in accepted roles
    def __call__(self, roles: List[str] = Depends(validate)) -> bool:
        accepted = [r for r in roles if r in self.allowed_roles]
        if len(accepted) > 0:
            return True
        else:
            raise HTTPException(status_code=403)


# Validators responsible for check authentication and authorization for the different routes
# allowed roles will have permission using the routes where validators are passed as dependencies
admin_validator = RemoteAuthValidator(allowed_roles=settings.admin_routes_roles)
scientist_validator = RemoteAuthValidator(allowed_roles=settings.scientist_routes_roles)
engineer_validator = RemoteAuthValidator(allowed_roles=settings.engineer_routes_roles)
common_validator = RemoteAuthValidator(allowed_roles=settings.common_routes_roles)
